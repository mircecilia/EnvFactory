"""CPU tests of production environment repairs; no verifier normalization."""
import hashlib
import unittest
from types import SimpleNamespace as NS
from envs.tools.GitHubServer import GitHubServer
from envs.tools.GorillaFileSystem import Directory
from envs.tools.WhatsApp import WhatsAppAPI
from envs.tools.MetMuseum import MetMuseumAPI,MuseumObject
from envs.tools.AirDNA import AirDNAAPI
from envs.tools.EtsyServer import EtsyServer
from envs.tools.TickTick import TickTickAPI
from envs.tools.TravelBooking import TravelBookingAPI


class DeterminismTest(unittest.TestCase):
    def test_airport_codes_unique_and_sorted(self):
        api=TravelBookingAPI()
        api.airport_city_map={'z':'SFO','a':'JFK','b':'SFO'}
        first=api.list_airports()
        api.airport_city_map=dict(reversed(list(api.airport_city_map.items())))
        self.assertEqual(first,api.list_airports())
        self.assertEqual(first,{'airport_codes':['JFK','SFO']})

    def test_github_hash_contract(self):
        api=GitHubServer();api.current_time='fixed'
        expected=int.from_bytes(hashlib.sha256(b'o/r/17/body').digest(),'big')%1000000
        self.assertEqual(api.add_issue_comment('o','r',17,'body')['comment_id'],expected)
        self.assertEqual(api.create_pull_request_review('o','r',17,'body','APPROVE')['review_id'],expected)

    def test_directory_unordered_entries(self):
        d=Directory('root');d.contents={'z':None,'a':None}
        self.assertEqual(d._list_contents(),['a','z'])

    def test_whatsapp_collection_and_limit(self):
        api=WhatsAppAPI()
        def chat(phone):return NS(phone=phone,archived=False,name=phone,unread_count=0,last_message='',last_message_time='')
        api.chats={'z':chat('z'),'a':chat('a')}
        first=api.whatsapp_list_chats(limit=1)
        api.chats=dict(reversed(list(api.chats.items())))
        self.assertEqual(first,api.whatsapp_list_chats(limit=1))
        self.assertEqual(first['chats'][0]['phone'],'a')

    def test_museum_identical_under_map_permutation(self):
        api=MetMuseumAPI();api.objects={'2':MuseumObject(objectId='2',title='test'),'1':MuseumObject(objectId='1',title='test')}
        first=api.search_museum_objects('test');api.objects=dict(reversed(list(api.objects.items())))
        self.assertEqual(first,api.search_museum_objects('test'))
        self.assertEqual(first['objectIds'],['1','2'])

    def test_markets_identical_under_map_permutation(self):
        api=AirDNAAPI();api.markets={k:NS(market_id=k,market_name='test',state='TX',market_type='metro') for k in ('z','a')}
        first=api.search_markets('test');api.markets=dict(reversed(list(api.markets.items())))
        self.assertEqual(first,api.search_markets('test'))

    def test_etsy_stable_limit_and_image_order_preserved(self):
        api=EtsyServer()
        def listing(i):return NS(listing_id=i,title='test',description='test',tags=[],price=NS(amount=10,divisor=1,currency='USD'),quantity=1,shop=NS(shop_name='s'),images=[NS(url='z'),NS(url='a')])
        api.listings={2:listing(2),1:listing(1)}
        first=api.search_listings('test',None,None,None,1);api.listings=dict(reversed(list(api.listings.items())))
        self.assertEqual(first,api.search_listings('test',None,None,None,1))
        self.assertEqual(first['results'][0]['listing_id'],1)
        self.assertEqual(first['results'][0]['images'],['z','a'])

    def test_ticktick_preserves_declared_column_order(self):
        api=TickTickAPI();api.projects={'p':NS(name='P')};api.project_columns={'p':['z','a']}
        api.tasks={k:NS(project_id='p',model_dump=lambda k=k:{'taskId':k}) for k in ('z','a')}
        api.columns={k:NS(project_id='p',model_dump=lambda k=k:{'column_id':k}) for k in ('a','z')}
        value=api.ticktick_get_project_with_data('p')
        self.assertEqual([x['taskId'] for x in value['tasks']],['a','z'])
        self.assertEqual([x['column_id'] for x in value['columns']],['z','a'])


if __name__=='__main__':unittest.main()
